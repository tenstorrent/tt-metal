# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
from ttml.trainers import FullyAsyncGRPOTrainer, FullyAsyncRolloutBatch


class _Completer:
    def __init__(self, batches):
        self.batches = list(batches)
        self.published = []
        self.started = None
        self.closed = False

    def start_async_rollouts(self, prompt_batches, *, initial_version):
        self.started = (list(prompt_batches), initial_version)

    def await_async_rollout(self):
        return self.batches.pop(0)

    def publish_weights(self, version):
        self.published.append(version)

    def close_async_rollouts(self):
        self.closed = True


def _batch(index, *, generations=2):
    prompts = [[index, 9], [index, 10]]
    completions = [[index, row] for row in range(len(prompts) * generations)]
    return FullyAsyncRolloutBatch(
        prompts=prompts,
        extra_columns={"answer": [f"a{index}-0", f"a{index}-1"]},
        completions=completions,
        behavior_version=index,
        behavior_logprobs=[[-0.1, -0.2] for _ in completions],
        group_id=f"batch-{index}",
    )


def _trainer(batches, *, iterations=1):
    trainer = FullyAsyncGRPOTrainer.__new__(FullyAsyncGRPOTrainer)
    trainer.completer = _Completer(batches)
    trainer.callbacks = []
    trainer.config = SimpleNamespace(num_generations=2, num_iterations=iterations)
    trainer.reward_funcs = []
    trainer._reward_func_names = []
    trainer._prompts = [[0], [1], [2], [3]]
    trainer._extra_dataset_columns = {"answer": ["x0", "x1", "x2", "x3"]}
    trainer._generation_batch_prompts = 2
    trainer.metrics = {}
    trainer.events = []

    trainer._setup = lambda: None
    trainer._reset_step_metrics = lambda: None
    trainer._publish_step_metrics = lambda: None
    trainer._maybe_checkpoint = lambda: None
    trainer._compute_rewards = lambda p, c, x: np.zeros(len(c), dtype=np.float32)
    trainer._compute_advantages = lambda rewards: rewards
    trainer._optimize = lambda p, c, a: trainer.events.append(("optimize", p, c))
    trainer._apply_gradients = lambda: trainer.events.append(("step",))
    return trainer


def test_fully_async_trainer_consumes_returned_prompts_and_publishes_each_update():
    returned = [_batch(7), _batch(8)]
    trainer = _trainer(returned, iterations=2)
    trainer.train()

    assert trainer.completer.started[1] == 0
    assert len(trainer.completer.started[0]) == 2
    assert trainer.completer.published == [1, 2, 3, 4]
    assert trainer.completer.closed
    optimized_prompts = [event[1] for event in trainer.events if event[0] == "optimize"]
    assert optimized_prompts[0] == [[7, 9], [7, 9], [7, 10], [7, 10]]
    assert trainer.last_rollout_batch.behavior_version == 8
    assert trainer.last_rollout_batch.behavior_logprobs[0] == [-0.1, -0.2]


def test_rollout_batch_rejects_token_logprob_misalignment(expect_error):
    batch = _batch(0)
    bad = FullyAsyncRolloutBatch(
        prompts=batch.prompts,
        extra_columns=batch.extra_columns,
        completions=batch.completions,
        behavior_version=0,
        behavior_logprobs=[[0.0]] * len(batch.completions),
        group_id="bad",
    )
    with expect_error(ValueError, "behavior logprobs"):
        bad.validate(2)
