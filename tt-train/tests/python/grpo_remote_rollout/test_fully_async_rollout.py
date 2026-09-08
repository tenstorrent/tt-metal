# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from threading import Thread

from ttml.trainers import FullyAsyncRolloutBatch

from utils.fully_async_rollout import FullyAsyncRolloutClient, FullyAsyncRolloutWorker
from utils.rollout_engine import RolloutOutput
from utils.rollout_transport import create_in_memory_rollout_transports


class _SenderBridge:
    def __init__(self):
        self.versions = []

    def publish(self, version, weights):
        self.versions.append((version, weights))

    def close(self):
        pass


class _ReceiverBridge:
    def __init__(self):
        self.updates = [type("Update", (), {"version": 0})()]

    def wait(self):
        return self.updates.pop(0)

    def poll(self):
        return self.updates.pop(0) if self.updates else None

    def materialize(self, update):
        return [{"version": update.version}]


class _Worker:
    def __init__(self):
        self.applied = []

    def update_weights(self, weights):
        self.applied.append(weights)

    def generate(self, prompts, **kwargs):
        return RolloutOutput.from_sequences(
            [[prompt[0], 99] for prompt in prompts],
            [[-0.25, -0.5] for _ in prompts],
        )


def test_bounded_stream_roundtrips_prompts_columns_versions_and_logprobs():
    pair = create_in_memory_rollout_transports(capacity=1)
    sender = _SenderBridge()
    receiver = _ReceiverBridge()
    worker = _Worker()
    client = FullyAsyncRolloutClient(
        transport=pair.trainer,
        weight_bridge=sender,
        weight_export=lambda: {"real": "weights"},
        num_generations=2,
    )
    service = FullyAsyncRolloutWorker(
        transport=pair.worker,
        weight_bridge=receiver,
        worker=worker,
        max_new_tokens=2,
    )
    thread = Thread(target=service.serve_forever)
    thread.start()
    client.start([([[1], [2]], {"answer": ["one", "two"]})], initial_version=0)
    batch = client.receive()
    client.close()
    thread.join(timeout=2)

    assert isinstance(batch, FullyAsyncRolloutBatch)
    assert batch.prompts == [[1], [2]]
    assert batch.extra_columns == {"answer": ["one", "two"]}
    assert batch.completions == [[1, 99], [1, 99], [2, 99], [2, 99]]
    assert batch.behavior_logprobs == [[-0.25, -0.5]] * 4
    assert batch.behavior_version == 0
    assert sender.versions == [(0, {"real": "weights"})]
    assert worker.applied == [[{"version": 0}]]
    assert not thread.is_alive()
