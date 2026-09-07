# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for token/logprob rollout assembly."""

from utils.rollout_output_collector import RolloutOutputCollector, sampled_token_logprobs


def test_collector_keeps_token_logprobs_aligned_across_eos_and_padding():
    collector = RolloutOutputCollector(
        batch_size=4,
        active_batch_size=3,
        stop_token_ids=[2],
        stop_at_eos=True,
    )

    collector.add_step([10, 20, 2, 99], [-0.1, -0.2, -0.3, -9.9])
    collector.add_step([11, 2, 30, 99], [-1.1, -1.2, -1.3, -9.9])
    collector.add_step([2, 21, 31, 99], [-2.1, -2.2, -2.3, -9.9])

    output = collector.finish()
    assert output.tokens == ((10, 11), (20,), ())
    assert output.logprobs == ((-0.1, -1.1), (-0.2,), ())


def test_collector_can_retain_stop_tokens_when_eos_handling_is_disabled():
    collector = RolloutOutputCollector(
        batch_size=1,
        active_batch_size=1,
        stop_token_ids=[2],
        stop_at_eos=False,
    )
    collector.add_step([2], [-0.5])
    assert collector.finish().tokens == ((2,),)
    assert collector.finish().logprobs == ((-0.5,),)


def test_sampled_logprobs_accept_scalar_and_topk_formats(expect_error):
    assert sampled_token_logprobs([7, 8], [-0.7, -0.8]) == [-0.7, -0.8]
    assert sampled_token_logprobs(
        [7, 8],
        (
            [[-0.1, -0.7], [-0.8, -0.2]],
            [[3, 7], [8, 4]],
        ),
    ) == [-0.7, -0.8]

    with expect_error(RuntimeError, "did not return"):
        sampled_token_logprobs([7], None)
    with expect_error(ValueError, "absent from top-k"):
        sampled_token_logprobs([7], ([[-0.1]], [[3]]))
