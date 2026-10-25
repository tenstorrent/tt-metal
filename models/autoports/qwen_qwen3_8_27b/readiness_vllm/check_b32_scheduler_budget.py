"""Host-only real scheduler admission probe using plugin test cache fixture.

Run from plugin root with PYTHONPATH=tests/host-stubs:. . Uses full-attention
host cache allocator fixture, not production hybrid allocation or hardware.
"""
import json
from tests.test_block_scheduler import _scheduler, _request

results = []
for budget in (8192, 131072):
    scheduler = _scheduler(output_width=1, max_model_len=262144, max_num_seqs=32)
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.max_num_batched_tokens = budget
    scheduler.scheduler_config.long_prefill_token_threshold = 4096
    scheduler.max_num_scheduled_tokens = budget
    scheduler._prefill_chunk_alignment = 4096
    for i in range(32):
        scheduler.add_request(_request(8, request_id=f'probe-{i}', prompt_length=4130))
    result = scheduler.schedule()
    counts = list(result.num_scheduled_tokens.values())
    assert len(counts) == budget // 4096, counts
    assert set(counts) == {4096}, counts
    results.append(dict(budget=budget, admitted=len(counts), chunk_tokens=counts))
print(json.dumps(dict(status='pass', results=results,
                     limitation='Real TT/upstream scheduler, host full-attention test allocator; not hybrid device validation')))
